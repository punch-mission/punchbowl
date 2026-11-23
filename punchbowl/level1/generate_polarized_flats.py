"""Build sequence-resolved WFI stellar-response and tensor-gain calibrations.

This self-contained module uses installed PUNCHbowl APIs, without experiment
module imports or sys.path changes. Run as a script to measure response maps
and then generate final gains, or call either public stage independently.
The editable parameters below control the scientific measurement and fitting.
"""

import csv
import gc
import itertools
import json
import multiprocessing as mp
import os
import re
import warnings
from bisect import bisect_left, bisect_right
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import reproject
from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.time import Time
from astropy.wcs import WCS
from matplotlib import pyplot as plt
from numpy.polynomial.chebyshev import chebval2d, chebvander2d
from scipy.ndimage import percentile_filter
from tqdm.auto import tqdm

from punchbowl.data import NormalizedMetadata, load_ndcube_from_fits
from punchbowl.data.punch_io import get_base_file_name
from punchbowl.data.punchcube import PUNCHCube
from punchbowl.data.wcs import calculate_helio_wcs_from_celestial
from punchbowl.level1.alignment import find_star_coordinates
from punchbowl.level2.preprocess import preprocess_trefoil_inputs

# PARAMETERS: use the same folder layout as run_star_background_server.py.
INPUT_ROOT = Path("/mnt/archive/soc/data/1")
INPUT_GLOB_PATTERN = "P[MPZ]{wfi}/2025/09/**/*v0m.fits"  # Relative to INPUT_ROOT.
OUTPUT_ROOT = Path("/home/soc/polarized_flatfield_test_20261030")
WFI_SELECTION = "all"
FILE_VERSION = None
MAXIMUM_TRIPLET_SPAN_S = 180.0
PREFER_HEADER_TIME = True
START_DAY = 1  # One-based UTC day index: 1 is the earliest discovered day.
END_DAY = 1
FIRST_TRIPLET_PER_WFI = 0
TRIPLET_STRIDE = 3  # 1: every M/Z/P triplet; 2: every other, within each WFI/sequence.
MAXIMUM_TRIPLETS_PER_WFI = None
CROTA_MATCH_TOLERANCE_DEG = 1.0  # PM/PZ/PP must share one roll position.
MAXIMUM_CROTA_DWELL_GAP_S = 360.0  # The two MZP triplets are about four minutes apart.
SEQUENCES = (1, 2)

# PARAMETERS: detector preprocessing and star measurement.
ALPHAS_FILE = None
IMAGE_MASKS = [None, None, None]
TRIM_EDGES_PX = 0
COALIGN_SCALE_FACTOR = 1.4
STAR_MEASUREMENT_MODE = "percentile_filtered"
TILE_SIZE_PX = 128
TILE_STEP_PX = TILE_SIZE_PX // 2
APERTURE_RADIUS_PX = 2.0
BACKGROUND_FILTER_SIZE_PX = 31
DETECTION_THRESHOLD_SIGMA = 5.0
SATURATION_LIMIT = float("inf")

# PARAMETERS: ensemble regression and scheduling.
MINIMUM_POINTS_PER_TILE = 10
MAX_STARS_PER_TRIPLET_PER_TILE = None
SIGMA_CLIP_ITERATIONS = 3
SIGMA_CLIP_LIMIT = 4.0
MAX_WORKERS = 0
MEMORY_GB_PER_WORKER = 4.0
RESERVE_MEMORY_GB = 32.0
SAVE_MEASUREMENT_CHECKPOINTS = True  # Save valid arrays before pooled regression.
OVERWRITE_OUTPUT = True

CHANNELS = ("M", "Z", "P")
HEADER_TIME_KEYS = ('DATE-OBS', 'DATEOBS', 'DATE-BEG', 'DATE-END', 'MJD-OBS', 'JD-OBS', 'MJD')
TENSOR_OVERWRITE_OUTPUT = False

# PARAMETERS: final clear-flat multiplication (third stage).
OLD_FLAT_DIR = OUTPUT_ROOT / "clear_flats"  # Path to clear flats per WFI.
FLAT_OBSERVATION_DATE = "2025-04-01T00:00:00"
AVERAGE_SEQUENCE_GAINS = False  # False: 18 sequence products; True: nine means.
FLAT_OVERWRITE_OUTPUT = False

DIRECT_WFIS = (1, 2, 3)
DIRECT_SEQUENCES = (1, 2)  # Written as seq-1 then seq-2 within each WFI.
DIRECT_CHANNELS = ("M", "Z", "P")
DIRECT_INPUT_FILENAME = "multiplicative_star_response.fits"
DIRECT_INPUT_HDU = "GAIN_SLOPE"

# Retain the image size and PUNCH filename settings of the earlier generator.
DIRECT_OUTPUT_SIZE = 2048
DIRECT_FILE_VERSION = "0m"
DIRECT_COALIGN_SCALE_FACTOR = 1.4
DIRECT_NORMALIZE_CENTER = True

# Angles are counterclockwise from the vertical reference direction in a
# detector image displayed with increasing array row upward.
DIRECT_AXIS_ANGLE_DEG = {"M": 30.0, "Z": 90.0, "P": -30.0}

# Quality cuts and statistical weights used in the comparison notebook.
DIRECT_MIN_POINTS_PER_TILE = 100
DIRECT_MIN_GAIN_CORRELATION = 0.95
DIRECT_MIN_VALID_TILE_FRACTION = 0.5
DIRECT_WEIGHT_LIMITS = (0.1, 10.0)

# Smooth non-separable Chebyshev model in polarizer-aligned coordinates.
DIRECT_ETA_ORDER = 3
DIRECT_ZETA_ORDER = 3
DIRECT_TENSOR_RIDGE = 0.03
DIRECT_HUBER_ITERATIONS = 12
DIRECT_HUBER_K = 1.5



def get_file_type(file_path, file_types):
    """Checks if the file matches any of the target types."""
    name = Path(file_path).name
    for t_type in file_types:
        if f'_{t_type}_' in name:
            return t_type
    return None



def get_obs_time(file_path, prefer_header=True):
    """Read observation time from FITS headers, falling back to its filename."""
    if prefer_header:
        # PUNCH image metadata live in extension 1, while some other FITS files
        # keep observation time in the primary header.  Check both before using
        # the lower-precision timestamp embedded in the filename.
        for extension in (0, 1):
            try:
                header = fits.getheader(file_path, extension)
                for key in HEADER_TIME_KEYS:
                    if key not in header:
                        continue
                    value = header[key]
                    if key.startswith('MJD'):
                        return Time(float(value), format='mjd')
                    if key.startswith('JD'):
                        return Time(float(value), format='jd')
                    return Time(value)
            except Exception:
                continue

    name = Path(file_path).name
    match = re.search(r'(20\d{12})', name)
    if match:
        return Time.strptime(match.group(1), '%Y%m%d%H%M%S')

    raise ValueError(f'Could not determine observation time for {file_path}')



def build_records(files, file_types, prefer_header=True, read_quality_metadata=False):
    """Read chronological frame records, retaining CROTA and OUTLIER for grouping."""
    records = []
    skipped = []

    for index, file_path in enumerate(files):
        f_type = get_file_type(file_path, file_types)
        if f_type is None:
            continue

        try:
            obs_time = get_obs_time(file_path, prefer_header=prefer_header)
            if read_quality_metadata:
                # Level-1 PUNCH metadata are stored with the image in HDU 1.
                header = fits.getheader(file_path, 1)
                crota = float(header['CROTA'])
                outlier = int(header['OUTLIER'])
                if not np.isfinite(crota):
                    raise ValueError('CROTA is not finite')
        except Exception as exc:
            skipped.append((index, file_path, str(exc)))
            continue

        record = {
            'index': index,
            'file': file_path,
            'type': f_type,
            'time': obs_time,
            'unix': float(obs_time.unix),
        }
        if read_quality_metadata:
            record['crota'] = crota
            record['outlier'] = outlier
        records.append(record)

    records.sort(key=lambda record: record['unix'])
    return records, skipped



def group_files(files, file_types=('PM1', 'PZ1', 'PP1'), max_span_s=180,
                prefer_header=True, unique_files=True, maximum_crota_spread_deg=None):
    """
    Generalized grouping. Works for pairs, triplets, quads, etc., 
    based on the tuple passed into `file_types`.
    """
    if maximum_crota_spread_deg is not None and maximum_crota_spread_deg <= 0:
        raise ValueError('maximum_crota_spread_deg must be positive')
    records, skipped = build_records(
        files, file_types, prefer_header=prefer_header,
        read_quality_metadata=maximum_crota_spread_deg is not None,
    )
    
    # Group records by their type
    grouped = {
        f_type: sorted(
            [record for record in records if record['type'] == f_type],
            key=lambda record: record['unix']
        )
        for f_type in file_types
    }

    missing = [f_type for f_type, group in grouped.items() if len(group) == 0]
    if missing:
        raise ValueError(f'Missing files for: {missing}')

    # Designate the first type as the "anchor" to set our time windows
    anchor_type = file_types[0]
    other_types = file_types[1:]
    
    # Pre-extract unix times for the 'other' types for fast binary searching
    other_times = {
        t_type: np.array([record['unix'] for record in grouped[t_type]])
        for t_type in other_types
    }
    
    candidates = []

    # Iterate over the anchor type
    for anchor_record in grouped[anchor_type]:
        left = anchor_record['unix'] - max_span_s
        right = anchor_record['unix'] + max_span_s

        # Find all valid records for the remaining types within the time window
        matches_per_type = []
        for t_type in other_types:
            t_array = other_times[t_type]
            t_records = grouped[t_type]
            
            idx_left = bisect_left(t_array, left)
            idx_right = bisect_right(t_array, right)
            matches_per_type.append(t_records[idx_left:idx_right])
            
        # If any of the required types have no matches in this window, skip early
        if any(len(m) == 0 for m in matches_per_type):
            continue

        # Generate every possible combination of matches using itertools.product
        # This replaces the hardcoded, nested `for` loops
        for other_combo in itertools.product(*matches_per_type):
            combo = (anchor_record,) + other_combo
            times = np.array([record['unix'] for record in combo])
            span_s = float(times.max() - times.min())

            if span_s > max_span_s:
                continue

            if maximum_crota_spread_deg is not None:
                angles = np.array([record['crota'] for record in combo])
                # Circular differences keep +179 and -179 degrees adjacent.
                differences = (angles - angles[0] + 180.0) % 360.0 - 180.0
                if np.max(np.abs(differences)) > maximum_crota_spread_deg:
                    continue

            candidate = {
                'indices': [record['index'] for record in combo],
                'files': [record['file'] for record in combo],
                'types': [record['type'] for record in combo],
                'times': [record['time'].isot for record in combo],
                'span_s': span_s,
                'center_unix': float(times.mean()),
            }
            if maximum_crota_spread_deg is not None:
                candidate['crota_deg'] = float((angles[0] + np.mean(differences) + 180.0) % 360.0 - 180.0)
                candidate['outlier_flags'] = tuple(record['outlier'] for record in combo)
            candidates.append(candidate)

    # Sort primarily by tightest span, then by time
    candidates.sort(key=lambda candidate: (candidate['span_s'], candidate['center_unix']))

    if not unique_files:
        return candidates, records, skipped

    selected = []
    used_indices = set()
    for candidate in candidates:
        if any(index in used_indices for index in candidate['indices']):
            continue
        selected.append(candidate)
        used_indices.update(candidate['indices'])

    return selected, records, skipped



def selected_wfi_numbers(wfi_selection: str | int) -> list[int]:
    """Interpret one WFI number or the string ``all``."""
    if isinstance(wfi_selection, str) and wfi_selection.lower() == "all":
        return [1, 2, 3]

    try:
        wfi_number = int(wfi_selection)
    except (TypeError, ValueError) as exc:
        raise ValueError("WFI_SELECTION must be 1, 2, 3, or 'all'") from exc

    if wfi_number not in (1, 2, 3):
        raise ValueError("WFI_SELECTION must be 1, 2, 3, or 'all'")
    return [wfi_number]



def discover_mzp_triplets(
    input_root: str | Path,
    wfi_selection: str | int,
    file_version: str | None,
    maximum_triplet_span_s: float,
    prefer_header_time: bool,
    maximum_crota_spread_deg: float | None = None,
    maximum_dwell_gap_s: float = 360.0,
    input_glob_pattern: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Find unique PM/PZ/PP combinations for one WFI or all three WFIs.

    Matching is delegated to the local ``group_files`` function.  The
    returned triplets are reordered chronologically after that function performs
    its tightest-span-first unique-file selection.
    """
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")
    if maximum_triplet_span_s <= 0:
        raise ValueError("maximum_triplet_span_s must be positive")
    if maximum_dwell_gap_s <= 0:
        raise ValueError("maximum_dwell_gap_s must be positive")

    triplets = []
    discovery_rows = []
    for wfi_number in selected_wfi_numbers(wfi_selection):
        # Archive layout: PMn/PZn/PPn/YYYY/MM/DD; retain the earlier WFI-n
        # layout for callers that have not supplied an archive search pattern.
        wfi_dir = input_root / f"WFI-{wfi_number}"
        if input_glob_pattern is not None:
            pattern = input_glob_pattern.format(wfi=wfi_number)
            if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise ValueError("input_glob_pattern must be relative to input_root")
            files = sorted(path for path in input_root.glob(pattern) if path.is_file())
            discovery_directory = input_root
        elif wfi_dir.is_dir():
            files = sorted(wfi_dir.rglob("*.fits"))
            discovery_directory = wfi_dir
        else:
            files = sorted(input_root.glob(f"P[MPZ]{wfi_number}/**/*.fits"))
            discovery_directory = input_root
        print(f"Found {len(files)} WFI{wfi_number} files!")
        if file_version is not None:
            version_text = file_version if file_version.startswith("v") else f"v{file_version}"
            files = [path for path in files if path.stem.endswith(f"_{version_text}")]
        print(f'Number of files: {len(files)} for WFI-{wfi_number}')
        if not files:
            raise FileNotFoundError(
                f"No WFI-{wfi_number} FITS matched in {input_root}; "
                f"pattern={input_glob_pattern!r}, version={file_version!r}"
            )
        file_types = (f"PM{wfi_number}", f"PZ{wfi_number}", f"PP{wfi_number}")
        grouped_triplets, records, skipped = group_files(
            [str(path) for path in files],
            file_types=file_types,
            max_span_s=maximum_triplet_span_s,
            prefer_header=prefer_header_time,
            unique_files=True,
            maximum_crota_spread_deg=maximum_crota_spread_deg,
        )
        grouped_triplets.sort(key=lambda triplet: triplet["center_unix"])

        # Label the two exposures of each fixed-CROTA dwell before any quality
        # rejection.  A missing or outlier first exposure must not make the
        # second exposure appear to be sequence 1.
        if maximum_crota_spread_deg is not None:
            dwell_groups = []
            current_group = []
            for triplet_index, triplet in enumerate(grouped_triplets):
                if current_group:
                    previous = grouped_triplets[current_group[-1]]
                    angle_difference = (
                        (triplet["crota_deg"] - previous["crota_deg"] + 180.0) % 360.0 - 180.0
                    )
                    time_difference = triplet["center_unix"] - previous["center_unix"]
                    if abs(angle_difference) > maximum_crota_spread_deg or time_difference > maximum_dwell_gap_s:
                        dwell_groups.append(current_group)
                        current_group = []
                current_group.append(triplet_index)
            if current_group:
                dwell_groups.append(current_group)
            for dwell_index, group in enumerate(dwell_groups):
                for position, triplet_index in enumerate(group, start=1):
                    grouped_triplets[triplet_index]["dwell_index"] = dwell_index
                    grouped_triplets[triplet_index]["sequence"] = position if len(group) == 2 else None

        for wfi_triplet_index, triplet in enumerate(grouped_triplets):
            files_by_type = {
                file_type: Path(file_path)
                for file_type, file_path in zip(triplet["types"], triplet["files"], strict=True)
            }
            center_time = Time(triplet["center_unix"], format="unix", scale="utc")
            triplets.append(
                {
                    "wfi": wfi_number,
                    "wfi_triplet_index": wfi_triplet_index,
                    "center_time": center_time.isot,
                    "center_id": center_time.strftime("%Y%m%dT%H%M%S"),
                    "span_s": float(triplet["span_s"]),
                    "center_unix": float(triplet["center_unix"]),
                    "pm_path": files_by_type[f"PM{wfi_number}"],
                    "pz_path": files_by_type[f"PZ{wfi_number}"],
                    "pp_path": files_by_type[f"PP{wfi_number}"],
                    "channel_times": tuple(triplet["times"]),
                    "crota_deg": triplet.get("crota_deg"),
                    "outlier_flags": triplet.get("outlier_flags"),
                    "dwell_index": triplet.get("dwell_index"),
                    "sequence": triplet.get("sequence"),
                }
            )

        if maximum_crota_spread_deg is not None:
            for triplet in triplets:
                if triplet["wfi"] != wfi_number:
                    continue
                if any(flag != 0 for flag in triplet["outlier_flags"]):
                    triplet["selection_reason"] = "nonzero_outlier"
                elif triplet["sequence"] is None:
                    triplet["selection_reason"] = "ambiguous_crota_dwell"
                else:
                    triplet["selection_reason"] = "accepted"

        discovery_rows.append(
            {
                "wfi": wfi_number,
                "directory": str(discovery_directory),
                "input_glob_pattern": input_glob_pattern,
                "candidate_fits_files": len(files),
                "timed_records": len(records),
                "skipped_files": len(skipped),
                "skipped_file_details": [
                    {"path": str(file_path), "reason": reason}
                    for _, file_path, reason in skipped
                ],
                "unique_triplets": len(grouped_triplets),
                "accepted_triplets": sum(
                    triplet.get("selection_reason") == "accepted"
                    for triplet in triplets if triplet["wfi"] == wfi_number
                ) if maximum_crota_spread_deg is not None else None,
                "file_version": file_version,
                "maximum_triplet_span_s": maximum_triplet_span_s,
            }
        )

    triplets.sort(key=lambda triplet: (triplet["center_time"], triplet["wfi"]))
    return triplets, discovery_rows



def select_triplet_range(
    triplets: list[dict],
    first_triplet_per_wfi: int,
    maximum_triplets_per_wfi: int | None,
    triplet_stride: int = 1,
) -> list[dict]:
    """Apply per-WFI input range, then stride independently within each sequence.

    Start/count define the original chronological range BEFORE subsampling.
    A stride of two retains every other M/Z/P triplet within each sequence,
    preserving both exposures of a CROTA dwell. Unlabelled runs stride per WFI.
    """
    if isinstance(triplet_stride, bool) or not isinstance(triplet_stride, int) or triplet_stride < 1:
        raise ValueError("triplet_stride must be a positive integer")
    if first_triplet_per_wfi < 0:
        raise ValueError("first_triplet_per_wfi cannot be negative")
    if maximum_triplets_per_wfi is not None and maximum_triplets_per_wfi <= 0:
        raise ValueError("maximum_triplets_per_wfi must be positive or None")

    selected = []
    for wfi_number in sorted({triplet["wfi"] for triplet in triplets}):
        wfi_triplets = [triplet for triplet in triplets if triplet["wfi"] == wfi_number]
        stop = None if maximum_triplets_per_wfi is None else first_triplet_per_wfi + maximum_triplets_per_wfi
        in_range = wfi_triplets[first_triplet_per_wfi:stop]
        sequences = dict.fromkeys(triplet.get("sequence") for triplet in in_range)
        for sequence in sequences:
            group = [triplet for triplet in in_range if triplet.get("sequence") == sequence]
            selected.extend(group[::triplet_stride])
    selected.sort(key=lambda triplet: (triplet["center_time"], triplet["wfi"]))
    return selected



def load_ordered_mzp(pm_path: str | Path, pz_path: str | Path, pp_path: str | Path) -> list[PUNCHCube]:
    """Load one PM/PZ/PP triplet and verify that it comes from one observatory."""
    cubes = [load_ndcube_from_fits(str(path)) for path in (pm_path, pz_path, pp_path)]
    expected_typecodes = ("PM", "PZ", "PP")
    actual_typecodes = tuple(cube.meta["TYPECODE"].value for cube in cubes)
    if actual_typecodes != expected_typecodes:
        raise ValueError(f"Expected PM/PZ/PP inputs; found {actual_typecodes}")

    obscodes = {cube.meta["OBSCODE"].value for cube in cubes}
    if len(obscodes) != 1:
        raise ValueError(f"The M/Z/P triplet spans multiple observatories: {sorted(obscodes)}")

    print(f"Loaded PM/PZ/PP shapes: {[cube.data.shape for cube in cubes]}")
    print(f"Observatory: WFI{cubes[0].meta['OBSCODE'].value}")
    return cubes



def copy_mzp_cubes(cubes: list[PUNCHCube]) -> list[PUNCHCube]:
    """Make independent double-precision data, uncertainty, WCS, and metadata copies.

    ``reproject_adaptive`` 0.19, used by the notebook's ``punchenv`` kernel,
    requires double-precision input/output buffers during L1 M/Z/P co-alignment.
    Keeping these arrays as float64 also matches ``load_ndcube_from_fits``'s
    default dtype.
    """
    copied = []
    for cube in cubes:
        copied.append(
            cube.replace(
                data=np.array(cube.data, dtype=np.float64, copy=True),
                uncertainty=StdDevUncertainty(
                    np.array(cube.uncertainty.array, dtype=np.float64, copy=True)
                ),
                wcs=cube.wcs.deepcopy(),
                celestial_wcs=(cube.celestial_wcs.deepcopy() if cube.celestial_wcs is not None else None),
                meta=deepcopy(cube.meta),
            )
        )
    return copied



def coalign_mzp_to_z_frame(mzp_cubes: list[PUNCHCube], scale_factor: float) -> list[PUNCHCube]:
    """Co-align an L1 M/Z/P triplet to the distortion-free Z-channel frame."""
    target_celestial_wcs = mzp_cubes[1].celestial_wcs.deepcopy()
    target_celestial_wcs.cpdis1 = None
    target_celestial_wcs.cpdis2 = None
    target_celestial_wcs.wcs.cdelt /= scale_factor
    target_celestial_wcs.wcs.crpix *= scale_factor
    target_size = int(target_celestial_wcs.array_shape[0] * scale_factor)
    target_shape = (target_size, target_size)
    target_helio_wcs = calculate_helio_wcs_from_celestial(
        target_celestial_wcs,
        mzp_cubes[1].meta.astropy_time,
        target_shape,
    )

    aligned_cubes = []
    for cube in mzp_cubes:
        output_array = np.empty((2, *target_shape), dtype=np.float64)
        reproject.reproject_adaptive(
            (
                np.stack((cube.data, cube.uncertainty.array), axis=0),
                cube.celestial_wcs,
            ),
            target_celestial_wcs,
            target_shape,
            output_array=output_array,
            roundtrip_coords=False,
            return_footprint=False,
        )
        aligned_cubes.append(
            cube.replace(
                data=output_array[0],
                uncertainty=StdDevUncertainty(output_array[1]),
                celestial_wcs=target_celestial_wcs,
                wcs=target_helio_wcs,
            )
        )
    return aligned_cubes



def make_detection_image(aligned_data: np.ndarray, background_filter_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Form a common-channel image and its percentile-filter unsharp representation."""
    with warnings.catch_warnings():
        # Pixels outside the common co-aligned field legitimately contain NaN
        # in all three channels and are filled below before source detection.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        detection_image = np.nanmedian(aligned_data, axis=0)
    finite = np.isfinite(detection_image)
    if not np.any(finite):
        raise ValueError("The aligned M/Z/P images contain no finite common area")

    fill_value = np.nanmedian(detection_image[finite])
    detection_filled = np.where(finite, detection_image, fill_value).astype(np.float32)
    smooth_background = percentile_filter(
        detection_filled,
        percentile=50,
        size=background_filter_size,
        mode="nearest",
    )
    unsharp_image = detection_filled - smooth_background
    return detection_image, unsharp_image



def circular_aperture_sums(
    data: np.ndarray,
    uncertainty: np.ndarray,
    x_stars: np.ndarray,
    y_stars: np.ndarray,
    aperture_radius_px: float,
    saturation_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Measure raw circular-aperture sums in all three aligned channels."""
    radius_int = int(np.ceil(aperture_radius_px))
    dy, dx = np.mgrid[-radius_int:radius_int + 1, -radius_int:radius_int + 1]
    aperture_mask = dx**2 + dy**2 <= aperture_radius_px**2
    aperture_pixel_count = int(np.sum(aperture_mask))

    fluxes = np.full((len(x_stars), 3), np.nan, dtype=float)
    flux_uncertainties = np.full_like(fluxes, np.nan)
    valid = np.zeros(len(x_stars), dtype=bool)

    ny, nx = data.shape[-2:]
    for star_index, (x_star, y_star) in enumerate(zip(x_stars, y_stars, strict=True)):
        x_center = int(np.rint(x_star))
        y_center = int(np.rint(y_star))
        if (
            x_center - radius_int < 0
            or x_center + radius_int >= nx
            or y_center - radius_int < 0
            or y_center + radius_int >= ny
        ):
            continue

        data_cutout = data[
            :,
            y_center - radius_int:y_center + radius_int + 1,
            x_center - radius_int:x_center + radius_int + 1,
        ][:, aperture_mask]
        uncertainty_cutout = uncertainty[
            :,
            y_center - radius_int:y_center + radius_int + 1,
            x_center - radius_int:x_center + radius_int + 1,
        ][:, aperture_mask]

        all_pixels_valid = np.all(np.isfinite(data_cutout)) and np.all(np.isfinite(uncertainty_cutout))
        below_saturation = np.all(data_cutout < saturation_limit)
        if not all_pixels_valid or not below_saturation:
            continue

        fluxes[star_index] = np.sum(data_cutout, axis=1)
        flux_uncertainties[star_index] = np.sqrt(np.sum(uncertainty_cutout**2, axis=1))
        valid[star_index] = True

    return fluxes, flux_uncertainties, valid, aperture_pixel_count



def make_channel_percentile_residuals(
    data: np.ndarray,
    background_filter_size_px: int,
) -> np.ndarray:
    """Subtract each channel's local percentile surface before sampling stars."""
    residuals = np.full(data.shape, np.nan, dtype=np.float32)
    for channel_index in range(3):
        finite = np.isfinite(data[channel_index])
        if not np.any(finite):
            continue
        fill_value = np.nanmedian(data[channel_index, finite])
        channel_filled = np.where(finite, data[channel_index], fill_value).astype(np.float32)
        local_background = percentile_filter(
            channel_filled,
            percentile=50,
            size=background_filter_size_px,
            mode="nearest",
        )
        residuals[channel_index] = channel_filled - local_background
        residuals[channel_index, ~finite] = np.nan
    return residuals



def percentile_filtered_aperture_sums(
    data: np.ndarray,
    uncertainty: np.ndarray,
    channel_percentile_residuals: np.ndarray,
    x_stars: np.ndarray,
    y_stars: np.ndarray,
    aperture_radius_px: float,
    saturation_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sum percentile-filtered intensities in a small circular star aperture.

    The percentile surface has already removed the local neighboring background.
    Therefore this is an aperture sum with *no separate annulus-background
    subtraction*.  The annulus used by conventional aperture photometry is not
    measured here; the aperture contains the filtered stellar residual itself.
    """
    residual_sums, residual_uncertainties, _, aperture_pixel_count = circular_aperture_sums(
        channel_percentile_residuals,
        uncertainty,
        x_stars,
        y_stars,
        aperture_radius_px,
        np.inf,
    )
    # Saturation and edge validity are assessed on the original detector data,
    # not on the percentile residual (whose values are intentionally reduced).
    _, _, original_valid, _ = circular_aperture_sums(
        data,
        uncertainty,
        x_stars,
        y_stars,
        aperture_radius_px,
        saturation_limit,
    )
    return residual_sums, residual_uncertainties, original_valid, aperture_pixel_count



def make_tile_origins(image_length: int, tile_size: int, tile_step: int) -> np.ndarray:
    """Return tile starts while forcing the final tile to meet the image boundary."""
    if tile_size > image_length:
        return np.array([0], dtype=int)
    origins = np.arange(0, image_length - tile_size + 1, tile_step, dtype=int)
    final_origin = image_length - tile_size
    if origins[-1] != final_origin:
        origins = np.append(origins, final_origin)
    return np.unique(origins)



def available_memory_gb() -> float:
    """Read available RAM without requiring psutil on a server image."""
    try:
        with Path("/proc/meminfo").open() as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024**2
    except (FileNotFoundError, OSError, ValueError):
        pass
    return float("inf")



def choose_worker_count() -> int:
    """Bound process count by CPU and estimated WFI-triplet memory."""
    cpu_limit = max(1, (os.cpu_count() or 1) - 1)
    if MAX_WORKERS > 0:
        return min(MAX_WORKERS, cpu_limit)
    available_memory = available_memory_gb()
    # Unknown RAM is represented by infinity; do not pass it to ``int``.
    if not np.isfinite(available_memory):
        return cpu_limit
    memory_limit = int(max(1.0, (available_memory - RESERVE_MEMORY_GB) / MEMORY_GB_PER_WORKER))
    return max(1, min(cpu_limit, memory_limit))



def measure_triplet(job: tuple[dict, dict]) -> dict:
    """Preprocess, co-align, detect, and measure stars for one triplet."""
    triplet, config = job
    try:
        native_cubes = load_ordered_mzp(triplet["pm_path"], triplet["pz_path"], triplet["pp_path"])
        preprocessed = copy_mzp_cubes(native_cubes)
        preprocess_trefoil_inputs(
            preprocessed,
            config["image_masks"],
            config["trim_edges_px"],
            config["alphas_file"],
        )
        del native_cubes
        aligned_cubes = coalign_mzp_to_z_frame(preprocessed, config["coalign_scale_factor"])
        data = np.stack([cube.data for cube in aligned_cubes]).astype(np.float32, copy=False)
        uncertainty = np.stack([cube.uncertainty.array for cube in aligned_cubes]).astype(np.float32, copy=False)
        _, unsharp_image = make_detection_image(data, config["background_filter_size_px"])
        coordinates = find_star_coordinates(
            unsharp_image,
            saturation_limit=np.inf,
            max_distance_from_center=np.inf,
            background_size=max(8, config["background_filter_size_px"] // 2),
            detection_threshold=config["detection_threshold_sigma"],
        )
        if len(coordinates) == 0:
            raise ValueError("No stellar candidates detected")
        x_stars, y_stars = coordinates[:, 0], coordinates[:, 1]
        if config["star_measurement_mode"] == "raw_aperture":
            measurements, measurement_uncertainties, valid, _ = circular_aperture_sums(
                data, uncertainty, x_stars, y_stars, config["aperture_radius_px"], config["saturation_limit"]
            )
        else:
            residuals = make_channel_percentile_residuals(data, config["background_filter_size_px"])
            measurements, measurement_uncertainties, valid, _ = percentile_filtered_aperture_sums(
                data, uncertainty, residuals, x_stars, y_stars,
                config["aperture_radius_px"], config["saturation_limit"],
            )
        valid &= np.all(np.isfinite(measurements), axis=1) & np.all(measurements > 0, axis=1)
        x_stars = x_stars[valid].astype(np.float32)
        y_stars = y_stars[valid].astype(np.float32)
        measurements = measurements[valid].astype(np.float32)
        measurement_uncertainties = measurement_uncertainties[valid].astype(np.float32)
        if len(x_stars) == 0:
            raise ValueError("No positive finite stellar measurements")
        result = {
            "wfi": triplet["wfi"],
            "sequence": triplet["sequence"],
            "dwell_index": triplet["dwell_index"],
            "crota_deg": triplet["crota_deg"],
            "wfi_triplet_index": triplet["wfi_triplet_index"],
            "center_time": triplet["center_time"],
            "x": x_stars,
            "y": y_stars,
            "measurements": measurements,
            "measurement_uncertainties": measurement_uncertainties,
            "image_shape": data.shape[-2:],
            "n_measurements": len(x_stars),
            "source_paths": [str(triplet[key]) for key in ("pm_path", "pz_path", "pp_path")],
            "status": "complete",
            "error": "",
        }
        del aligned_cubes, preprocessed, data, uncertainty, unsharp_image
        gc.collect()
        return result
    except Exception as exc:
        gc.collect()
        return {
            "wfi": triplet["wfi"], "sequence": triplet["sequence"],
            "dwell_index": triplet["dwell_index"], "crota_deg": triplet["crota_deg"],
            "wfi_triplet_index": triplet["wfi_triplet_index"],
            "center_time": triplet["center_time"], "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }



def fit_line(x: np.ndarray, y: np.ndarray, force_zero: bool) -> tuple[float, float, float, int]:
    """Fit ``y = slope*x + intercept`` after robust residual clipping."""
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite].astype(float)
    y = y[finite].astype(float)
    if len(x) < MINIMUM_POINTS_PER_TILE or np.ptp(x) <= 0:
        return np.nan, np.nan, np.nan, len(x)
    keep = np.ones(len(x), dtype=bool)
    slope = intercept = np.nan
    for _ in range(SIGMA_CLIP_ITERATIONS + 1):
        xx, yy = x[keep], y[keep]
        if len(xx) < MINIMUM_POINTS_PER_TILE or np.ptp(xx) <= 0:
            return np.nan, np.nan, np.nan, len(xx)
        if force_zero:
            slope = float(np.sum(xx * yy) / np.sum(xx * xx))
            intercept = 0.0
        else:
            slope, intercept = np.polyfit(xx, yy, 1)
            slope, intercept = float(slope), float(intercept)
        residuals = y - (slope * x + intercept)
        center = np.nanmedian(residuals[keep])
        scale = 1.4826 * np.nanmedian(np.abs(residuals[keep] - center))
        if not np.isfinite(scale) or scale == 0:
            break
        keep = np.abs(residuals - center) <= SIGMA_CLIP_LIMIT * scale
    xx, yy = x[keep], y[keep]
    if len(xx) < MINIMUM_POINTS_PER_TILE or np.ptp(xx) <= 0:
        return np.nan, np.nan, np.nan, len(xx)
    if force_zero:
        slope, intercept = float(np.sum(xx * yy) / np.sum(xx * xx)), 0.0
    else:
        slope, intercept = np.polyfit(xx, yy, 1)
        slope, intercept = float(slope), float(intercept)
    correlation = float(np.corrcoef(xx, yy)[0, 1]) if np.ptp(yy) > 0 else np.nan
    return slope, intercept, correlation, int(np.sum(keep))



def build_response_maps(results: list[dict], wfi_number: int, sequence: int) -> tuple[dict, list[dict]]:
    """Pool stars into overlapping tiles and calculate requested maps."""
    valid_results = [
        result for result in results
        if result.get("status") == "complete" and result["wfi"] == wfi_number
        and result["sequence"] == sequence
    ]
    if not valid_results:
        raise ValueError(f"No successful triplets for WFI-{wfi_number} sequence-{sequence}")
    ny, nx = valid_results[0]["image_shape"]
    x_origins = make_tile_origins(nx, TILE_SIZE_PX, TILE_STEP_PX)
    y_origins = make_tile_origins(ny, TILE_SIZE_PX, TILE_STEP_PX)
    x_centers = x_origins.astype(float) + (TILE_SIZE_PX - 1) / 2
    y_centers = y_origins.astype(float) + (TILE_SIZE_PX - 1) / 2
    shape = (len(y_origins), len(x_origins))
    maps = {name: np.full((3, *shape), np.nan, dtype=np.float32) for name in (
        "ratio_slope", "ratio_intercept", "ratio_r", "gain_no_offset", "gain_slope", "gain_intercept", "gain_r", "n_points", "n_triplets", "n_raw", "n_gain", "n_ratio"
    )}
    rows = []
    for yi, y0 in enumerate(y_origins):
        for xi, x0 in enumerate(x_origins):
            x_all, measurements_all, triplet_ids = [], [], []
            for result in valid_results:
                inside = ((result["x"] >= x0) & (result["x"] < x0 + TILE_SIZE_PX)
                          & (result["y"] >= y0) & (result["y"] < y0 + TILE_SIZE_PX))
                indices = np.flatnonzero(inside)
                if MAX_STARS_PER_TRIPLET_PER_TILE is not None and len(indices) > MAX_STARS_PER_TRIPLET_PER_TILE:
                    rss = np.sqrt(np.sum(result["measurements"][indices] ** 2, axis=1))
                    indices = indices[np.argsort(rss)[-MAX_STARS_PER_TRIPLET_PER_TILE:]]
                if len(indices):
                    x_all.append(result["x"][indices])
                    measurements_all.append(result["measurements"][indices])
                    triplet_ids.extend([result["wfi_triplet_index"]] * len(indices))
            if not measurements_all:
                continue
            measurements = np.concatenate(measurements_all)
            rss = np.sqrt(np.sum(measurements**2, axis=1))
            n_triplets = len(set(triplet_ids))
            for channel_index, channel in enumerate(CHANNELS):
                intensity = measurements[:, channel_index]
                ratio = intensity / rss
                ratio_slope, ratio_intercept, ratio_r, n_ratio = fit_line(intensity, ratio, False)
                gain_slope, gain_intercept, gain_r, n_gain = fit_line(rss, intensity, False)
                gain_no_offset, _, _, _ = fit_line(rss, intensity, True)
                maps["ratio_slope"][channel_index, yi, xi] = ratio_slope
                maps["ratio_intercept"][channel_index, yi, xi] = ratio_intercept
                maps["ratio_r"][channel_index, yi, xi] = ratio_r
                maps["gain_no_offset"][channel_index, yi, xi] = gain_no_offset
                maps["gain_slope"][channel_index, yi, xi] = gain_slope
                maps["gain_intercept"][channel_index, yi, xi] = gain_intercept
                maps["gain_r"][channel_index, yi, xi] = gain_r
                maps["n_raw"][channel_index, yi, xi] = len(measurements)
                maps["n_gain"][channel_index, yi, xi] = n_gain
                maps["n_ratio"][channel_index, yi, xi] = n_ratio
                maps["n_points"][channel_index, yi, xi] = n_gain
                maps["n_triplets"][channel_index, yi, xi] = n_triplets
                rows.append({
                    "wfi": wfi_number, "sequence": sequence,
                    "tile_y": yi, "tile_x": xi, "x_center": x_centers[xi], "y_center": y_centers[yi],
                    "channel": channel, "n_points": n_gain, "n_triplets": n_triplets,
                    "n_raw": len(measurements), "n_gain": n_gain, "n_ratio": n_ratio,
                    "ratio_slope": ratio_slope, "ratio_intercept": ratio_intercept, "ratio_r": ratio_r,
                    "gain_no_offset": gain_no_offset, "gain_slope": gain_slope, "gain_intercept": gain_intercept, "gain_r": gain_r,
                })
    maps["aligned_image_shape"] = (ny, nx)
    maps["x_tile_centers"] = x_centers
    maps["y_tile_centers"] = y_centers
    return maps, rows



def write_star_measurements(path: Path, results: list[dict], wfi_number: int, sequence: int) -> int:
    """Write every valid observation; legacy star_id is a unique observation ID."""
    fields = ["star_id", "wfi", "sequence", "dwell_index", "crota_deg", "triplet_index", "center_time", "x", "y",
              "I_M", "I_Z", "I_P", "r_M", "r_Z", "r_P"]
    count = 0
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            if (result.get("status") != "complete" or result["wfi"] != wfi_number
                    or result["sequence"] != sequence):
                continue
            measurements = result["measurements"]
            rss = np.sqrt(np.sum(measurements**2, axis=1))
            ratios = measurements / rss[:, None]
            for index, star_id in enumerate(result["star_ids"]):
                writer.writerow({
                    "star_id": int(star_id), "wfi": wfi_number, "sequence": sequence,
                    "dwell_index": result["dwell_index"], "crota_deg": result["crota_deg"],
                    "triplet_index": result["wfi_triplet_index"],
                    "center_time": result["center_time"],
                    "x": float(result["x"][index]), "y": float(result["y"][index]),
                    "I_M": float(measurements[index, 0]), "I_Z": float(measurements[index, 1]),
                    "I_P": float(measurements[index, 2]), "r_M": float(ratios[index, 0]),
                    "r_Z": float(ratios[index, 1]), "r_P": float(ratios[index, 2]),
                })
                count += 1
    return count



def write_response_fits(path: Path, maps: dict, wfi_number: int, sequence: int,
                        n_triplets: int, overwrite: bool) -> None:
    """Write regression maps and tile coordinates as a multi-extension FITS."""
    primary = fits.PrimaryHDU()
    primary.header["METHOD"] = "STAR-MULTIPLICATIVE-REGRESSION"
    primary.header["MODEL"] = "q_i=I_i/RSS versus I_i"
    primary.header["GAINMOD"] = "I_i versus RSS, no intercept"
    primary.header["WFI"] = wfi_number
    primary.header["SEQNUM"] = sequence
    primary.header["NTRIP"] = n_triplets
    primary.header["CROTATOL"] = CROTA_MATCH_TOLERANCE_DEG
    primary.header["DWELLGAP"] = MAXIMUM_CROTA_DWELL_GAP_S
    primary.header["TILESIZE"] = TILE_SIZE_PX
    primary.header["TILESTEP"] = TILE_STEP_PX
    primary.header["MEASMODE"] = STAR_MEASUREMENT_MODE
    primary.header["ALNSCALE"] = COALIGN_SCALE_FACTOR
    primary.header["STRIDE"] = TRIPLET_STRIDE
    primary.header["IDMODE"] = "OBSERVATION"
    primary.header["ALNNY"], primary.header["ALNNX"] = maps["aligned_image_shape"]
    primary.header["COORDFRM"] = "UNDISTORTED-Z"
    hdus = [primary]
    for name in ("ratio_slope", "ratio_intercept", "ratio_r", "gain_no_offset", "gain_slope", "gain_intercept", "gain_r", "n_points", "n_triplets", "n_raw", "n_gain", "n_ratio"):
        hdus.append(fits.ImageHDU(maps[name], name=name.upper()[:16]))
    hdus.extend([fits.ImageHDU(maps["x_tile_centers"], name="TILE_X"), fits.ImageHDU(maps["y_tile_centers"], name="TILE_Y")])
    fits.HDUList(hdus).writeto(path, overwrite=overwrite, checksum=True)



def save_response_plot(path: Path, maps: dict, wfi_number: int, sequence: int) -> None:
    """Plot the requested slope/intercept/R maps for all three channels."""
    figure, axes = plt.subplots(3, 3, figsize=(16, 14), squeeze=False)
    columns = (("ratio_slope", "Ratio slope"), ("ratio_intercept", "Ratio intercept"), ("ratio_r", "Pearson r"))
    for channel_index, channel in enumerate(CHANNELS):
        for column_index, (name, title) in enumerate(columns):
            image = axes[channel_index, column_index].imshow(maps[name][channel_index], origin="lower", cmap="coolwarm" if name != "ratio_r" else "viridis")
            axes[channel_index, column_index].set_title(f"WFI-{wfi_number} sequence-{sequence} {channel}: {title}")
            figure.colorbar(image, ax=axes[channel_index, column_index], shrink=0.8)
    figure.suptitle("Pooled stellar multiplicative-response regression")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)



def save_gain_plot(path: Path, maps: dict, wfi_number: int, sequence: int) -> None:
    """Plot the dimensionless no-intercept gain fit for reviewer interpretation."""
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
    for channel_index, channel in enumerate(CHANNELS):
        image = axes[0, channel_index].imshow(maps["gain_no_offset"][channel_index], origin="lower", cmap="coolwarm")
        axes[0, channel_index].set_title(f"WFI-{wfi_number} sequence-{sequence} {channel}: gain (Iᵢ versus RSS)")
        figure.colorbar(image, ax=axes[0, channel_index], shrink=0.8)
    figure.suptitle("No-intercept multiplicative gain diagnostic")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)



def summarize_response_maps(maps: dict, wfi_number: int, sequence: int) -> dict:
    """Summarize unity departure, intercept scale, and fit quality per channel."""
    summary = {"wfi": wfi_number, "sequence": sequence, "channels": {}}
    for channel_index, channel in enumerate(CHANNELS):
        gain = maps["gain_no_offset"][channel_index]
        intercept = maps["ratio_intercept"][channel_index]
        correlation = maps["ratio_r"][channel_index]
        finite_gain = gain[np.isfinite(gain)]
        finite_intercept = intercept[np.isfinite(intercept)]
        finite_correlation = correlation[np.isfinite(correlation)]
        summary["channels"][channel] = {
            "finite_tiles": int(len(finite_gain)),
            "gain_median": float(np.nanmedian(finite_gain)) if len(finite_gain) else None,
            "gain_std": float(np.nanstd(finite_gain)) if len(finite_gain) else None,
            "gain_median_departure_percent": float(100 * (np.nanmedian(finite_gain) - 1)) if len(finite_gain) else None,
            "ratio_intercept_median": float(np.nanmedian(finite_intercept)) if len(finite_intercept) else None,
            "ratio_intercept_abs_median": float(np.nanmedian(np.abs(finite_intercept))) if len(finite_intercept) else None,
            "ratio_r_median": float(np.nanmedian(finite_correlation)) if len(finite_correlation) else None,
        }
    return summary



def analyze_stellar_response(
    input_root: str | Path,
    output_root: str | Path,
    input_glob_pattern: str | None = INPUT_GLOB_PATTERN,
) -> Path:
    """Measure stars and fit each CROTA-dwell sequence independently.

    Parameters
    ----------
    input_root : str or Path
        Archive root containing PMn/PZn/PPn subdirectories, or WFI-n folders.
    output_root : str or Path
        Run directory; response products are saved under multiplicative_response.
    input_glob_pattern : str or None
        Relative glob with a {wfi} placeholder. None enables layout detection.

    Returns
    -------
    Path
        Response directory consumed by generate_tensor_calibrations.
    """
    if START_DAY < 1 or END_DAY < START_DAY:
        raise ValueError("Require 1 <= START_DAY <= END_DAY (inclusive UTC days)")
    output_root = Path(output_root) / "multiplicative_response"
    output_root.mkdir(parents=True, exist_ok=True)
    existing_products = list(output_root.glob("WFI-*/sequence-*/*")) + list(output_root.glob("measurements/WFI-*/sequence-*/*.npz"))
    if (output_root / "run_parameters.json").exists():
        existing_products.append(output_root / "run_parameters.json")
    if existing_products and not OVERWRITE_OUTPUT:
        raise FileExistsError(f"Response products already exist in {output_root}; choose a fresh OUTPUT_ROOT or explicitly set OVERWRITE_OUTPUT=True")
    triplets, discovery_rows = discover_mzp_triplets(
        Path(input_root), WFI_SELECTION, FILE_VERSION, MAXIMUM_TRIPLET_SPAN_S,
        PREFER_HEADER_TIME, CROTA_MATCH_TOLERANCE_DEG, MAXIMUM_CROTA_DWELL_GAP_S,
        input_glob_pattern=input_glob_pattern,
    )
    if not triplets:
        raise RuntimeError("No triplets discovered")

    # Day 1 is the earliest discovered UTC calendar date.
    day_1 = datetime.fromisoformat(
        min(triplet["center_time"] for triplet in triplets)
    ).replace(hour=0, minute=0, second=0, microsecond=0)

    start_time = day_1 + timedelta(days=START_DAY - 1)
    stop_time = day_1 + timedelta(days=END_DAY)

    # An exclusive stop includes the whole final day.
    triplets = [
        triplet for triplet in triplets
        if start_time <= datetime.fromisoformat(triplet["center_time"]) < stop_time
    ]

    print(f"Selected UTC interval: {start_time} <= t < {stop_time}")
    print(f"Triplets in selected interval: {len(triplets)}")
    
    triplets = select_triplet_range(triplets, FIRST_TRIPLET_PER_WFI, MAXIMUM_TRIPLETS_PER_WFI, TRIPLET_STRIDE)
    if not triplets:
        raise RuntimeError("No triplets matched the requested selection")
    accepted_triplets = [triplet for triplet in triplets if triplet["selection_reason"] == "accepted"]
    if not accepted_triplets:
        raise RuntimeError("No complete, OUTLIER-free CROTA-dwell pairs matched the selection")
    config = {
        "image_masks": [None if path is None else str(path) for path in IMAGE_MASKS],
        "alphas_file": None if ALPHAS_FILE is None else str(ALPHAS_FILE), "trim_edges_px": TRIM_EDGES_PX,
        "coalign_scale_factor": COALIGN_SCALE_FACTOR, "star_measurement_mode": STAR_MEASUREMENT_MODE,
        "aperture_radius_px": APERTURE_RADIUS_PX, "background_filter_size_px": BACKGROUND_FILTER_SIZE_PX,
        "detection_threshold_sigma": DETECTION_THRESHOLD_SIGMA, "saturation_limit": SATURATION_LIMIT,
    }
    print(
        f"Selected {len(accepted_triplets)} of {len(triplets)} triplets after CROTA/OUTLIER checks; "
        f"workers={choose_worker_count()}; available RAM={available_memory_gb():.1f} GB"
    )
    context = mp.get_context("spawn")
    results = []
    n_complete = n_failed = 0
    with context.Pool(processes=choose_worker_count(), maxtasksperchild=1) as pool:
        with tqdm(total=len(accepted_triplets), desc="Stellar response", unit="triplet") as progress:
            for result in pool.imap_unordered(measure_triplet, [(triplet, config) for triplet in accepted_triplets]):
                if SAVE_MEASUREMENT_CHECKPOINTS and result["status"] == "complete":
                    checkpoint_root = output_root / "measurements" / f"WFI-{result['wfi']}" / f"sequence-{result['sequence']}"
                    checkpoint_root.mkdir(parents=True, exist_ok=True)
                    checkpoint_meta = {key: result[key] for key in (
                        "wfi", "sequence", "dwell_index", "crota_deg", "wfi_triplet_index", "center_time", "image_shape", "source_paths"
                    )}
                    checkpoint_meta["measurement_config"] = config
                    np.savez(
                        checkpoint_root / f"triplet-{result['wfi_triplet_index']:06d}.npz",
                        x=result["x"], y=result["y"], measurements=result["measurements"],
                        measurement_uncertainties=result["measurement_uncertainties"],
                        metadata=json.dumps(checkpoint_meta),
                    )
                results.append(result)
                n_complete += result["status"] == "complete"
                n_failed += result["status"] == "failed"
                progress.update(1)
                progress.set_postfix(complete=n_complete, failed=n_failed)
                tqdm.write(
                    f"{len(results)} out of {len(accepted_triplets)} total triplets done; "
                    f"WFI-{result['wfi']} sequence-{result['sequence']}: {result['status']}; "
                    f"complete={n_complete}, failed={n_failed} {result.get('error', '')}"
                )
    successful = [result for result in results if result.get("status") == "complete"]
    # Persistent star identity is unnecessary for I_channel versus RSS regression.
    # Retain all detections; later days must never delete earlier observations.
    observation_offset = 0
    successful.sort(key=lambda result: (result["wfi"], result["sequence"], result["wfi_triplet_index"]))
    for result in successful:
        n_observations = len(result["x"])
        result["star_ids"] = np.arange(observation_offset, observation_offset + n_observations)
        observation_offset += n_observations
    print(f"Successful triplets: {len(successful)} / {len(accepted_triplets)}; retained all {observation_offset} valid stellar observations")
    result_by_triplet = {
        (result["wfi"], result["wfi_triplet_index"]): result for result in results
    }
    
    audit_fields = [
        "wfi", "wfi_triplet_index", "dwell_index", "sequence", "center_time",
        "crota_deg", "pm_outlier", "pz_outlier", "pp_outlier", "decision",
        "processing_status", "processing_error", "n_measurements", "pm_path", "pz_path", "pp_path",
    ]
    with (output_root / "triplet_decisions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=audit_fields)
        writer.writeheader()
        for triplet in triplets:
            result = result_by_triplet.get((triplet["wfi"], triplet["wfi_triplet_index"]), {})
            flags = triplet["outlier_flags"]
            writer.writerow({
                "wfi": triplet["wfi"], "wfi_triplet_index": triplet["wfi_triplet_index"],
                "dwell_index": triplet["dwell_index"], "sequence": triplet["sequence"],
                "center_time": triplet["center_time"], "crota_deg": triplet["crota_deg"],
                "pm_outlier": flags[0], "pz_outlier": flags[1], "pp_outlier": flags[2],
                "decision": triplet["selection_reason"],
                "processing_status": result.get("status", "not_processed"),
                "processing_error": result.get("error", ""),
                "n_measurements": result.get("n_measurements", 0),
                "pm_path": triplet["pm_path"], "pz_path": triplet["pz_path"],
                "pp_path": triplet["pp_path"],
            })
    (output_root / "run_parameters.json").write_text(json.dumps({
        "input_root": str(input_root), "input_glob_pattern": input_glob_pattern,
        "start_day": START_DAY, "end_day": END_DAY,
        "output_root": str(output_root), "wfi_selection": WFI_SELECTION,
        "file_version": FILE_VERSION, "maximum_triplets_per_wfi": MAXIMUM_TRIPLETS_PER_WFI,
        "star_measurement_mode": STAR_MEASUREMENT_MODE, "aperture_radius_px": APERTURE_RADIUS_PX,
        "tile_size_px": TILE_SIZE_PX, "tile_step_px": TILE_STEP_PX, "minimum_points_per_tile": MINIMUM_POINTS_PER_TILE,
        "successful_triplets": len(successful), "failed_triplets": len(results) - len(successful),
        "triplet_stride": TRIPLET_STRIDE,
        "first_triplet_per_wfi": FIRST_TRIPLET_PER_WFI,
        "coalign_scale_factor": COALIGN_SCALE_FACTOR,
        "identity_mode": "unique_observation_no_cross_triplet_filtering",
        "retained_measurements": observation_offset,
        "save_measurement_checkpoints": SAVE_MEASUREMENT_CHECKPOINTS,
        "crota_match_tolerance_deg": CROTA_MATCH_TOLERANCE_DEG,
        "maximum_crota_dwell_gap_s": MAXIMUM_CROTA_DWELL_GAP_S,
        "accepted_triplets": len(accepted_triplets),
        "rejected_triplets": len(triplets) - len(accepted_triplets),
        "discovery": discovery_rows,
    }, indent=2) + "\n")
    for wfi_number in sorted({triplet["wfi"] for triplet in triplets}):
        for sequence in SEQUENCES:
            group = [
                result for result in successful
                if result["wfi"] == wfi_number and result["sequence"] == sequence
            ]
            if not group:
                print(f"WFI-{wfi_number} sequence-{sequence}: no successful triplets; no response file written")
                continue
            maps, rows = build_response_maps(successful, wfi_number, sequence)
            wfi_output = output_root / f"WFI-{wfi_number}" / f"sequence-{sequence}"
            wfi_output.mkdir(parents=True, exist_ok=True)
            response_path = wfi_output / "multiplicative_star_response.fits"
            if OVERWRITE_OUTPUT or not response_path.exists():
                write_response_fits(
                    response_path, maps, wfi_number, sequence, len(group), OVERWRITE_OUTPUT,
                )
            with (wfi_output / "multiplicative_star_response_tiles.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            n_star_measurements = write_star_measurements(
                wfi_output / "multiplicative_star_measurements.csv", successful, wfi_number, sequence
            )
            (wfi_output / "multiplicative_star_response_summary.json").write_text(
                json.dumps(summarize_response_maps(maps, wfi_number, sequence), indent=2) + "\n"
            )
            save_response_plot(wfi_output / "multiplicative_star_response.png", maps, wfi_number, sequence)
            save_gain_plot(wfi_output / "multiplicative_star_gain.png", maps, wfi_number, sequence)
            print(
                f"WFI-{wfi_number} sequence-{sequence}: wrote {len(rows)} tile fits, "
                f"{n_star_measurements} retained observations, and PNG products"
            )

    return output_root



def generate_tensor_calibrations(
    response_root: str | Path,
    final_output_root: str | Path,
    wfis: tuple[int, ...] = DIRECT_WFIS,
) -> list[Path]:
    """Fit positive smooth detector gains and write separate WFI/sequence products.

    Parameters
    ----------
    response_root : str or Path
        Directory containing WFI-n/sequence-n/multiplicative_star_response.fits.
    final_output_root : str or Path
        Destination for the 18 center-normalized 2048-square GM/GZ/GP products.
    wfis : tuple of int
        Instruments to process in the given order; each writes sequences 1 then 2.

    Returns
    -------
    list of Path
        Written products in WFI-major, sequence-1 then sequence-2 order.

    Notes
    -----
    Fits log(C) = sum(c_ij T_i(eta) T_j(zeta)) using clipped stellar-count
    weights, Huber reweighting and ridge penalty lambda*(i+j)**4.
    Tile coordinates map to native pixels as (aligned + 1)/ALNSCALE - 1.
    """
    DIRECT_ROOT = Path(response_root)
    FINAL_OUTPUT_ROOT = Path(final_output_root)
    DIRECT_WFIS = wfis
    if DIRECT_ETA_ORDER < 0 or DIRECT_ZETA_ORDER < 0:
        raise ValueError("Chebyshev orders must be non-negative")
    if DIRECT_TENSOR_RIDGE < 0:
        raise ValueError("DIRECT_TENSOR_RIDGE must be non-negative")
    if DIRECT_HUBER_ITERATIONS < 1:
        raise ValueError("DIRECT_HUBER_ITERATIONS must be at least one")
    if DIRECT_HUBER_K <= 0:
        raise ValueError("DIRECT_HUBER_K must be positive")
    if DIRECT_WEIGHT_LIMITS[0] <= 0 or DIRECT_WEIGHT_LIMITS[1] < DIRECT_WEIGHT_LIMITS[0]:
        raise ValueError("DIRECT_WEIGHT_LIMITS must be positive and increasing")


    calibration_time = datetime.now(UTC).replace(microsecond=0)
    written_paths = []

    # This ordering is deliberate: WFI-1 seq-1, WFI-1 seq-2, WFI-2 seq-1, ...
    input_groups = [
        (wfi, sequence)
        for wfi in DIRECT_WFIS
        for sequence in DIRECT_SEQUENCES
    ]

    # Validate the complete set before writing any calibration products.
    for wfi, sequence in input_groups:
        input_path = (
            DIRECT_ROOT
            / f"WFI-{wfi}"
            / f"sequence-{sequence}"
            / DIRECT_INPUT_FILENAME
        )
        if not input_path.is_file():
            raise FileNotFoundError(input_path)

    FINAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


    for group_number, (wfi, sequence) in enumerate(
        tqdm(input_groups, desc="Tensor calibrations", unit="response"), start=1
    ):
        input_path = (
            DIRECT_ROOT
            / f"WFI-{wfi}"
            / f"sequence-{sequence}"
            / DIRECT_INPUT_FILENAME
        )

        with fits.open(input_path, memmap=True) as source:
            required_hdus = (
                DIRECT_INPUT_HDU,
                "GAIN_R",
                "TILE_X",
                "TILE_Y",
            )
            missing_hdus = [name for name in required_hdus if name not in source]
            if missing_hdus:
                raise KeyError(f"{input_path}: missing HDUs {missing_hdus}")
            if "N_GAIN" not in source and "N_POINTS" not in source:
                raise KeyError(f"{input_path}: missing both N_GAIN and N_POINTS")

            slope_data = np.array(
                source[DIRECT_INPUT_HDU].data, dtype=np.float64, copy=True
            )
            gain_r = np.array(source["GAIN_R"].data, dtype=np.float64, copy=True)
            count_hdu = "N_GAIN" if "N_GAIN" in source else "N_POINTS"
            n_gain = np.array(source[count_hdu].data, dtype=np.float64, copy=True)
            tile_x = np.array(source["TILE_X"].data, dtype=np.float64, copy=True)
            tile_y = np.array(source["TILE_Y"].data, dtype=np.float64, copy=True)
            source_header = source[0].header.copy()
            aligned_scale = float(
                source_header.get("ALNSCALE", DIRECT_COALIGN_SCALE_FACTOR)
            )

        if not np.isfinite(aligned_scale) or aligned_scale <= 0:
            raise ValueError(f"{input_path}: invalid ALNSCALE={aligned_scale}")
        if source_header.get("SEQNUM") != sequence or source_header.get("WFI") != wfi:
            raise ValueError(f"{input_path}: WFI/SEQNUM does not match its directory")
        if slope_data.shape != (3, tile_y.size, tile_x.size):
            raise ValueError(
                f"{input_path}: {DIRECT_INPUT_HDU} shape {slope_data.shape} is "
                f"inconsistent with TILE_Y={tile_y.size}, TILE_X={tile_x.size}"
            )
        if gain_r.shape != slope_data.shape or n_gain.shape != slope_data.shape:
            raise ValueError(f"{input_path}: gain quality arrays have inconsistent shapes")
        if np.any(np.diff(tile_x) <= 0) or np.any(np.diff(tile_y) <= 0):
            raise ValueError(f"{input_path}: TILE_X and TILE_Y must increase monotonically")

        print(
            f"WFI-{wfi} sequence-{sequence}: extracted {DIRECT_INPUT_HDU} "
            f"{slope_data.shape}; support={count_hdu}; "
            f"TILE_X={tile_x.size}, TILE_Y={tile_y.size}"
        )

        center = (DIRECT_OUTPUT_SIZE - 1.0) / 2.0
        tile_y_grid, tile_x_grid = np.meshgrid(tile_y, tile_x, indexing="ij")

        # The response tiles are in the co-aligned frame.  Convert their one-based
        # FITS pixel centers back to native detector coordinates before fitting.
        x_native = (tile_x_grid + 1.0) / aligned_scale - 1.0
        y_native = (tile_y_grid + 1.0) / aligned_scale - 1.0
        if (
            np.min(x_native) < 0
            or np.max(x_native) >= DIRECT_OUTPUT_SIZE
            or np.min(y_native) < 0
            or np.max(y_native) >= DIRECT_OUTPUT_SIZE
        ):
            raise ValueError(
                f"{input_path}: tile coordinates do not map into the native detector grid"
            )

        x_sample = x_native - center
        y_sample = y_native - center
        output_row, output_column = np.indices(
            (DIRECT_OUTPUT_SIZE, DIRECT_OUTPUT_SIZE), dtype=np.float64
        )
        x_output = output_column - center
        y_output = output_row - center

        for channel_index, channel in enumerate(DIRECT_CHANNELS):
            observed = slope_data[channel_index]
            channel_r = gain_r[channel_index]
            channel_count = n_gain[channel_index]

            valid = (
                np.isfinite(observed)
                & (observed > 0)
                & np.isfinite(channel_count)
                & (channel_count >= DIRECT_MIN_POINTS_PER_TILE)
                & np.isfinite(channel_r)
                & (channel_r >= DIRECT_MIN_GAIN_CORRELATION)
            )
            valid_fraction = float(np.mean(valid))
            print(
                f"WFI-{wfi} sequence-{sequence} {channel}: "
                f"{np.count_nonzero(valid)}/{valid.size} quality tiles "
                f"({valid_fraction:.1%}); aligned scale={aligned_scale}"
            )
            if valid_fraction < DIRECT_MIN_VALID_TILE_FRACTION:
                raise ValueError(
                    f"WFI-{wfi} sequence-{sequence} {channel}: inadequate "
                    f"spatial coverage ({valid_fraction:.1%})"
                )

            angle = np.deg2rad(DIRECT_AXIS_ANGLE_DEG[channel])
            eta_sample_pixel = x_sample * np.cos(angle) + y_sample * np.sin(angle)
            zeta_sample_pixel = -x_sample * np.sin(angle) + y_sample * np.cos(angle)
            eta_output_pixel = x_output * np.cos(angle) + y_output * np.sin(angle)
            zeta_output_pixel = -x_output * np.sin(angle) + y_output * np.cos(angle)

            # Normalization to [-1, 1] gives the Chebyshev basis its stable domain.
            eta_scale = float(np.max(np.abs(eta_output_pixel)))
            zeta_scale = float(np.max(np.abs(zeta_output_pixel)))
            eta_sample = eta_sample_pixel / eta_scale
            zeta_sample = zeta_sample_pixel / zeta_scale
            eta_output = np.clip(eta_output_pixel / eta_scale, -1.0, 1.0)
            zeta_output = np.clip(zeta_output_pixel / zeta_scale, -1.0, 1.0)

            n_valid = int(np.count_nonzero(valid))
            n_coefficients = (DIRECT_ETA_ORDER + 1) * (DIRECT_ZETA_ORDER + 1)
            if n_valid < n_coefficients:
                raise ValueError(
                    f"WFI-{wfi} sequence-{sequence} {channel}: {n_valid} valid "
                    f"tiles cannot constrain {n_coefficients} coefficients"
                )

            design = chebvander2d(
                eta_sample[valid],
                zeta_sample[valid],
                [DIRECT_ETA_ORDER, DIRECT_ZETA_ORDER],
            ).reshape(n_valid, n_coefficients)
            target = np.log(observed[valid])

            # More stellar measurements provide a firmer gain estimate, but the
            # clipping prevents a handful of dense tiles from controlling the fit.
            count_scale = float(np.median(channel_count[valid]))
            if not np.isfinite(count_scale) or count_scale <= 0:
                raise RuntimeError(f"WFI-{wfi} sequence-{sequence} {channel}: invalid count scale")
            base_weight = np.clip(
                channel_count[valid] / count_scale,
                DIRECT_WEIGHT_LIMITS[0],
                DIRECT_WEIGHT_LIMITS[1],
            )

            degree_pairs = np.array(
                [
                    (eta_order, zeta_order)
                    for eta_order in range(DIRECT_ETA_ORDER + 1)
                    for zeta_order in range(DIRECT_ZETA_ORDER + 1)
                ],
                dtype=np.int64,
            )
            # Penalizing total degree suppresses small-scale lobes while retaining
            # a broad gradient.  The constant gain level is left unpenalized.
            ridge_diagonal = (
                DIRECT_TENSOR_RIDGE * np.sum(degree_pairs, axis=1, dtype=float) ** 4
            )
            ridge_diagonal[0] = 0.0

            robust_weight = np.ones(n_valid, dtype=np.float64)
            for iteration in range(DIRECT_HUBER_ITERATIONS):
                sqrt_weight = np.sqrt(base_weight * robust_weight)
                weighted_design = design * sqrt_weight[:, None]
                weighted_target = target * sqrt_weight
                coefficients = np.linalg.solve(
                    weighted_design.T @ weighted_design + np.diag(ridge_diagonal),
                    weighted_design.T @ weighted_target,
                )

                log_residual = target - design @ coefficients
                residual_center = float(np.median(log_residual))
                robust_scale = float(
                    1.4826 * np.median(np.abs(log_residual - residual_center))
                    + 1e-12
                )
                cutoff = DIRECT_HUBER_K * robust_scale
                absolute_residual = np.abs(log_residual - residual_center)
                robust_weight = np.minimum(
                    1.0, cutoff / np.maximum(absolute_residual, 1e-30)
                )

            coefficient_grid = coefficients.reshape(
                DIRECT_ETA_ORDER + 1, DIRECT_ZETA_ORDER + 1
            )
            fitted_samples = np.exp(
                chebval2d(eta_sample[valid], zeta_sample[valid], coefficient_grid)
            )
            sample_residual = observed[valid] - fitted_samples
            residual_rms = float(np.sqrt(np.mean(sample_residual**2)))
            residual_median = float(np.median(sample_residual))
            residual_mad = float(
                np.median(np.abs(sample_residual - residual_median))
            )

            # chebval2d evaluates the 2048 x 2048 surface without constructing a
            # very large (N_pixel x N_coefficient) design matrix.
            gain = np.exp(chebval2d(eta_output, zeta_output, coefficient_grid))
            center_value_before_normalization = float(
                gain[DIRECT_OUTPUT_SIZE // 2, DIRECT_OUTPUT_SIZE // 2]
            )
            if DIRECT_NORMALIZE_CENTER:
                if (
                    not np.isfinite(center_value_before_normalization)
                    or center_value_before_normalization <= 0
                ):
                    raise RuntimeError(
                        f"WFI-{wfi} sequence-{sequence} {channel}: invalid center gain"
                    )
                gain /= center_value_before_normalization
                # T_0(eta)T_0(zeta)=1, so this updates the stored model itself to
                # the same center-normalized surface written to the FITS file.
                coefficient_grid[0, 0] -= np.log(center_value_before_normalization)

            if not np.all(np.isfinite(gain)) or np.min(gain) <= 0:
                raise RuntimeError(
                    f"WFI-{wfi} sequence-{sequence} {channel}: "
                    "model is non-finite or non-positive"
                )
            gain = gain.astype(np.float32)

            metadata = NormalizedMetadata.load_template(f"G{channel}{wfi}", "1")
            # Sequence epochs differ by one second, preserving distinct PUNCH names
            # while the loop writes seq-1 before seq-2 for every WFI.
            metadata["DATE-OBS"] = (
                calibration_time + timedelta(seconds=sequence - 1)
            ).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
            metadata["FILEVRSN"] = DIRECT_FILE_VERSION
            metadata["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

            pixel_wcs = WCS(naxis=2)
            pixel_wcs.wcs.crpix = [(DIRECT_OUTPUT_SIZE + 1) / 2.0] * 2
            pixel_wcs.wcs.cdelt = [1.0, 1.0]
            pixel_wcs.wcs.crval = [0.0, 0.0]
            pixel_wcs.wcs.ctype = ["PIXEL", "PIXEL"]
            cube = PUNCHCube(data=gain, wcs=pixel_wcs, meta=metadata)
            header = cube.meta.to_fits_header(
                wcs=cube.wcs, write_celestial_wcs=False
            )
            filename = f"{get_base_file_name(cube)}.fits"

            header["FILENAME"] = filename
            header["CALTYPE"] = "SMOOTH-STELLAR-GAIN"
            header["FITMODEL"] = "CHEBYSHEV2D-LOG"
            header["SEQNUM"] = sequence
            header["NTRIP"] = int(source_header["NTRIP"])
            header["INSEQ"] = f"WFI-{wfi}/sequence-{sequence}"
            header["INFILE"] = input_path.name
            header["SRCDATE"] = str(source_header.get("DATE-OBS", "UNAVAILABLE"))
            header.add_history(f"Source: {input_path}")
            header["INPUTHDU"] = DIRECT_INPUT_HDU
            header["COUNTHDU"] = count_hdu
            header["INTERMED"] = "NONE"
            header["ANGCONV"] = "CCW-FROM-VERTICAL"
            header["YCONV"] = "UP=ARRAY-ROW"
            header["AXISDEG"] = float(DIRECT_AXIS_ANGLE_DEG[channel])
            header["ETAORDER"] = DIRECT_ETA_ORDER
            header["ZETORDER"] = DIRECT_ZETA_ORDER
            header["RIDGE"] = DIRECT_TENSOR_RIDGE
            header["HUBITER"] = DIRECT_HUBER_ITERATIONS
            header["HUBERK"] = DIRECT_HUBER_K
            header["WGHTMIN"] = DIRECT_WEIGHT_LIMITS[0]
            header["WGHTMAX"] = DIRECT_WEIGHT_LIMITS[1]
            header["FITRMS"] = residual_rms
            header["FITMAD"] = residual_mad
            header["NVALID"] = n_valid
            header["ALNSCALE"] = aligned_scale
            header["VALIDFR"] = valid_fraction
            header["MINPTS"] = DIRECT_MIN_POINTS_PER_TILE
            header["MINCORR"] = DIRECT_MIN_GAIN_CORRELATION
            header["XYMAP"] = "(aligned+1)/scale-1"
            header["NPIXELS"] = int(gain.size)
            header["OUTMASK"] = "NONE"
            header["CENTERN"] = bool(DIRECT_NORMALIZE_CENTER)
            header["CNTRRAW"] = center_value_before_normalization
            for eta_order in range(DIRECT_ETA_ORDER + 1):
                for zeta_order in range(DIRECT_ZETA_ORDER + 1):
                    header[f"TC{eta_order}{zeta_order}"] = float(
                        coefficient_grid[eta_order, zeta_order]
                    )

            hdu_data = fits.ImageHDU(
                data=cube.data, header=header, name="Primary data array"
            )
            provenance_column = fits.Column(
                name="provenance",
                format="A80",
                array=np.char.array(cube.meta.provenance),
            )
            hdu_provenance = fits.BinTableHDU.from_columns(
                fits.ColDefs([provenance_column])
            )
            hdu_provenance.name = "File provenance"

            output_path = FINAL_OUTPUT_ROOT / filename
            fits.HDUList(
                [fits.PrimaryHDU(), hdu_data, hdu_provenance]
            ).writeto(output_path, overwrite=TENSOR_OVERWRITE_OUTPUT, checksum=True)

            written_paths.append(output_path)
            maximum_location = np.unravel_index(np.argmax(gain), gain.shape)
            print(
                f"WFI-{wfi} sequence-{sequence} {channel}: wrote {output_path}; "
                f"center={gain[DIRECT_OUTPUT_SIZE // 2, DIRECT_OUTPUT_SIZE // 2]:.6f}; "
                f"range=({np.min(gain):.6f}, {np.max(gain):.6f}); "
                f"MAD={residual_mad:.6f}; maximum row,column={maximum_location}"
            )

        tqdm.write(
            f"{group_number} out of {len(input_groups)} response files done; "
            f"{group_number * len(DIRECT_CHANNELS)} calibration files written"
        )

    return written_paths


def generate_updated_flats(
    old_flat_dir: str | Path,
    gain_dir: str | Path,
    output_dir: str | Path,
    wfis: tuple[int, ...] = (1, 2, 3),
    file_version: str = "0m",
    observation_date: str = "2025-04-01T00:00:00",
    average_sequence_gains: bool = False,
    overwrite: bool = False,
    history_text: str = "Updated current flat with corrected gains across FOV.",
) -> list[Path]:
    """Multiply old flats by their matching stellar gains and write new flats.

    Parameters
    ----------
    old_flat_dir : str or Path
        One existing GM/GZ/GP flat for each selected WFI/channel.
    gain_dir : str or Path
        Two tensor gain products per WFI/channel, identified by SEQNUM 1 and 2.
    output_dir : str or Path
        Destination for the updated calibration products.
    wfis : tuple of int
        WFI processing order. Within each WFI, sequence 1 precedes sequence 2.
    file_version : str
        PUNCH version without the leading v.
    observation_date : str
        DATE-OBS validity epoch for both sequences. Filename seconds 01 and 02
        distinguish sequences while DATE-OBS remains fixed at this epoch.
    average_sequence_gains : bool
        False writes F_new,s = F_old * g_s (18 products for three WFIs).
        True writes F_new = F_old * (g_1 + g_2)/2 (nine products).
    overwrite : bool
        Allow replacement of existing output files; input replacement is refused.
    history_text : str
        FITS HISTORY describing the update; gain source paths are also recorded.

    Returns
    -------
    list of Path
        Updated flats in WFI/sequence/channel processing order.

    Notes
    -----
    Preserves the old flat HDUs, dtype, WCS, and masked pixels. Updates FILENAME,
    FILEVRSN, DATE, DATE-OBS, sequence provenance, and existing checksums.
    """
    OLD_FLAT_DIR = Path(old_flat_dir)
    GAIN_DIR = Path(gain_dir)
    OUTPUT_DIR = Path(output_dir)
    wfi_number = wfis
    DATA_HDU = "PRIMARY DATA ARRAY"
    FILE_VERSION = file_version
    OBSERVATION_DATE = observation_date
    AVERAGE_SEQUENCE_GAINS = average_sequence_gains
    OVERWRITE_OUTPUT = overwrite
    HISTORY_TEXT = history_text
    old_flats = sorted(OLD_FLAT_DIR.glob("*.fits"))
    gain_all = sorted(GAIN_DIR.glob("*.fits"))
    observation_filename_date = datetime.fromisoformat(OBSERVATION_DATE).strftime("%Y%m%d%H%M")

    # PM/PZ/PP science channels use GM/GZ/GP calibration products.
    # First identify seq-1 and seq-2 from SEQNUM rather than filename order.
    product_inputs = {}
    for wfi in wfi_number:
        for channel in CHANNELS:
            product = f"G{channel}{wfi}"
            flat_paths = [p for p in old_flats if p.name.split("_")[2] == product]
            gain_paths = [p for p in gain_all if p.name.split("_")[2] == product]
            if len(flat_paths) != 1 or len(gain_paths) != 2:
                raise ValueError(f"{product}: expected 1 old flat and 2 gains; found {len(flat_paths)} and {len(gain_paths)}")

            flat_path = flat_paths[0]
            with fits.open(flat_path, memmap=True) as hdus:
                flat_header = hdus[DATA_HDU].header
                flat_shape = hdus[DATA_HDU].shape
                flat_date_obs = str(flat_header["DATE-OBS"])
                if str(flat_header["TYPECODE"]).strip() + str(flat_header["OBSCODE"]).strip() != product:
                    raise ValueError(f"Product mismatch: {flat_path}")
                for key in ("FILEVRSN", "FILENAME", "DATE", "DATE-OBS"):
                    if key not in flat_header:
                        raise KeyError(f"{flat_path}: missing existing keyword {key}")

            gain_by_sequence = {}
            for gain_path in gain_paths:
                with fits.open(gain_path, memmap=True) as hdus:
                    gain_header = hdus[DATA_HDU].header
                    gain_product = str(gain_header["TYPECODE"]).strip() + str(gain_header["OBSCODE"]).strip()
                    if gain_product != product:
                        raise ValueError(f"Product mismatch: {gain_path}")
                    if hdus[DATA_HDU].shape != flat_shape:
                        raise ValueError(f"{gain_path}: shape {hdus[DATA_HDU].shape} differs from flat {flat_shape}")
                    sequence = int(gain_header["SEQNUM"])
                    if sequence not in (1, 2) or sequence in gain_by_sequence:
                        raise ValueError(f"{product}: invalid or duplicate sequence {sequence}")
                    gain_by_sequence[sequence] = (gain_path, str(gain_header["DATE-OBS"]))
            if set(gain_by_sequence) != {1, 2}:
                raise ValueError(f"{product}: expected sequence numbers 1 and 2")
            product_inputs[(wfi, channel)] = (
                product, flat_path, flat_date_obs, gain_by_sequence
            )

    flat_jobs = []
    if AVERAGE_SEQUENCE_GAINS:
        # Original behavior: one mean-gain output for each WFI/channel.
        for wfi in wfi_number:
            for channel in CHANNELS:
                product, flat_path, flat_date_obs, gain_by_sequence = product_inputs[(wfi, channel)]
                gain_paths = [gain_by_sequence[1][0], gain_by_sequence[2][0]]
                filename = flat_path.name.rsplit("_v", 1)[0] + f"_v{FILE_VERSION}.fits"
                output_path = OUTPUT_DIR / filename
                flat_jobs.append((product, 0, flat_path, gain_paths, output_path, OBSERVATION_DATE))
                print(f"{product}: {flat_path.name} x mean({gain_paths[0].name}, {gain_paths[1].name})")
    else:
        # Requested behavior: finish all seq-1 channels, then all seq-2 channels, per WFI.
        for wfi in wfi_number:
            for sequence in (1, 2):
                for channel in CHANNELS:
                    product, flat_path, flat_date_obs, gain_by_sequence = product_inputs[(wfi, channel)]
                    gain_path, _gain_date_obs = gain_by_sequence[sequence]
                    # DATE-OBS stays at 00 seconds; filename seconds 01/02 identify the sequence.
                    filename = (
                        f"PUNCH_L1_{product}_{observation_filename_date}"
                        f"{sequence:02d}_v{FILE_VERSION}.fits"
                    )
                    output_path = OUTPUT_DIR / filename
                    flat_jobs.append((product, sequence, flat_path, [gain_path], output_path, OBSERVATION_DATE))
                    print(f"{product} sequence-{sequence}: {flat_path.name} x {gain_path.name}")

    for product, sequence, flat_path, gain_paths, output_path, output_date_obs in flat_jobs:
        if output_path.resolve() in [p.resolve() for p in [flat_path] + gain_paths]:
            raise ValueError(f"Output would replace an input: {output_path}")
        if output_path.exists() and not OVERWRITE_OUTPUT:
            raise FileExistsError(output_path)

    print(f"Ready to write {len(flat_jobs)} flats to {OUTPUT_DIR}")


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written_flats = []
    for product, sequence, flat_path, gain_paths, output_path, output_date_obs in flat_jobs:
        # Copy the original HDUs so no template-only keywords or provenance HDUs appear.
        with fits.open(flat_path, memmap=False) as source:
            output_hdus = fits.HDUList([hdu.copy() for hdu in source])
        try:
            flat_hdu = output_hdus[DATA_HDU]
            old_flat = np.array(flat_hdu.data, dtype=np.float64, copy=True)
            old_dtype = flat_hdu.data.dtype

            gain_arrays = []
            for gain_path in gain_paths:
                with fits.open(gain_path, memmap=False) as gain_hdus:
                    gain = np.array(gain_hdus[DATA_HDU].data, dtype=np.float64, copy=True)
                if not np.all(np.isfinite(gain)):
                    raise ValueError(f"{product}: non-finite gain values in {gain_path}")
                gain_arrays.append(gain)

            if AVERAGE_SEQUENCE_GAINS:
                # F_new(x,y) = F_old(x,y) [g_1(x,y) + g_2(x,y)] / 2.
                gain_applied = np.mean(gain_arrays, axis=0)
            else:
                # F_new,seq(x,y) = F_old(x,y) g_seq(x,y).
                gain_applied = gain_arrays[0]
            new_flat = (old_flat * gain_applied).astype(old_dtype)
            flat_hdu.data = new_flat

            # These existing FITS fields are simple strings. Updating them
            # directly preserves every other old-flat metadata/WCS card.
            flat_hdu.header["FILEVRSN"] = FILE_VERSION
            flat_hdu.header["FILENAME"] = output_path.name
            flat_hdu.header["DATE"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
            flat_hdu.header["DATE-OBS"] = output_date_obs
            flat_hdu.header["GAINMODE"] = "AVERAGE" if AVERAGE_SEQUENCE_GAINS else "SEQUENCE"
            flat_hdu.header["SEQNUM"] = sequence
            flat_hdu.header["NGAINS"] = len(gain_paths)
            flat_hdu.header.add_history(HISTORY_TEXT)
            for gain_path in gain_paths:
                flat_hdu.header.add_history(f"Applied gain: {gain_path}")

            # Refresh existing checksum cards without adding new keyword names.
            for hdu in output_hdus:
                if "CHECKSUM" in hdu.header:
                    if "DATASUM" not in hdu.header:
                        raise KeyError(f"{flat_path}: CHECKSUM exists without DATASUM")
                    hdu.add_checksum()
                elif "DATASUM" in hdu.header:
                    hdu.add_datasum()
            output_hdus.writeto(output_path, overwrite=OVERWRITE_OUTPUT)
            written_flats.append(output_path)
            sequence_text = "mean" if AVERAGE_SEQUENCE_GAINS else f"sequence-{sequence}"
            print(f"{product} {sequence_text}: wrote {output_path.name}; shape={new_flat.shape}, dtype={new_flat.dtype}")
        finally:
            output_hdus.close()

    print(f"Saved {len(written_flats)} new flats to {OUTPUT_DIR}")

    return written_flats


def main() -> list[Path]:
    """Measure stellar response, fit tensor gains, and update existing flats."""
    response_root = analyze_stellar_response(INPUT_ROOT, OUTPUT_ROOT, INPUT_GLOB_PATTERN)
    generate_tensor_calibrations(
        response_root, OUTPUT_ROOT / "final_calibration",
        tuple(selected_wfi_numbers(WFI_SELECTION)),
    )

    return generate_updated_flats(
        OLD_FLAT_DIR, OUTPUT_ROOT / "final_calibration", OUTPUT_ROOT / "new_flats",
        wfis=tuple(selected_wfi_numbers(WFI_SELECTION)),
        file_version=DIRECT_FILE_VERSION,
        observation_date=FLAT_OBSERVATION_DATE,
        average_sequence_gains=AVERAGE_SEQUENCE_GAINS,
        overwrite=FLAT_OVERWRITE_OUTPUT,
    )


# The guard is required so spawned measurement workers import without rerunning.
if __name__ == "__main__":
    main()
