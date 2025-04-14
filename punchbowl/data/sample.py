"""
Module for downloading sample data files.

This module provides the following sample data files.  When a sample shortname
is accessed, the corresponding file is downloaded if needed.  All files can be
downloaded by calling :func:`~punchbowl.data.sample.download_all`.

Code adapted from SunPy.
"""
from pathlib import Path
from urllib.parse import urljoin

import parfive
from sunpy import log
from sunpy.util.config import _is_writable_dir
from sunpy.util.parfive_helpers import Downloader

_BASE_URLS = (
    "https://data.boulder.swri.edu/lowder/PUNCH/sample/"
)

_SAMPLE_DATA = {
    "PUNCH_DK4": "PUNCH_L0_DK4_20250414154536_v1.fits",
    "PUNCH_PAM": "PUNCH_L3_PAM_20240620000000.fits",
    "PUNCH_PAN": "PUNCH_L3_PAN_20240620000000.fits",
    "PUNCH_PNN": "PUNCH_L3_PNN_20240620000000.fits",
    "PUNCH_PTM": "PUNCH_L3_PTM_20240620000000.fits",
    "QUICKPUNCH_WQM": "PUNCH_LQ_WQM_20230704012000.fits",
    "QUICKPUNCH_NQN": "PUNCH_LQ_NQN_20230704022400.fits",
}

_SAMPLE_FILES = {v: k for k, v in _SAMPLE_DATA.items()}


def get_and_create_sample_dir() -> Path:
    """Get the config of download directory and create one if not present."""
    sample_dir = Path("PUNCH_SAMPLE").expanduser().resolve()
    if not _is_writable_dir(sample_dir):
        raise RuntimeError(f'Could not write to SunPy sample data directory="{sample_dir}"')

    return sample_dir


def _download_sample_data(base_url:str, sample_files:list, overwrite:bool) -> parfive.Results:
    """
    Download a list of files.

    Parameters
    ----------
    base_url : str
        Base URL for each file.
    sample_files : list of tuples
        List of tuples that are (URL_NAME, SAVE_NAME).
    overwrite : bool
        Will overwrite a file on disk if True.

    Returns
    -------
    `parfive.Results`
        Download results. Will behave like a list of files.

    """
    dl = Downloader(overwrite=overwrite, progress=True)

    for url_file_name, fname in sample_files:
        url = urljoin(base_url, url_file_name)
        dl.enqueue_file(url, filename=fname)

    return dl.download()


def _retry_sample_data(results: parfive.Results, new_url_base:str) -> parfive.Results:
    """
    Retry failed downloads.

    Parameters
    ----------
    results : `parfive.Results`
        Download results.
    new_url_base : str
        Base URL for each file.

    Returns
    -------
    `parfive`
        Download results. Will behave like a list of files.

    """
    dl = Downloader(overwrite=True, progress=True)

    for err in results.errors:
        file_name = err.url.split("/")[-1]
        log.debug(
            f"Failed to download {_SAMPLE_FILES[file_name]} from {err.url}: {err.exception}")
        # Update the url to a mirror and requeue the file.
        new_url = urljoin(new_url_base, file_name)
        log.debug(f"Attempting redownload of {_SAMPLE_FILES[file_name]} using {new_url}")
        dl.enqueue_file(new_url, filename=err.filepath_partial)

    extra_results = dl.download()

    # Make a new results object which contains all the successful downloads
    # from the previous results object and this retry, and all the errors from
    # this retry.
    new_results = results + extra_results
    new_results._errors = extra_results._errors  # noqa: SLF001
    return new_results


def _handle_final_errors(results:parfive.Results) -> None:
    for err in results.errors:
        file_name = err.url.split("/")[-1]
        log.debug(f"Failed to download {_SAMPLE_FILES[file_name]} from {err.url}: {err.exception}",
                  )
        log.error(
            f"Failed to download {_SAMPLE_FILES[file_name]} from all mirrors,"
            "the file will not be available.",
        )


def _get_sampledata_dir() -> Path:
    return Path(get_and_create_sample_dir())


def _get_sample_files(filename_list: list, no_download:bool=False, force_download:bool=False) -> list[Path]:
    """
    Return a list of disk locations for a list of sample data filenames, downloading sample data files as needed.

    Parameters
    ----------
    filename_list : `list` of `str`
        List of filenames for sample data
    no_download : `bool`
        If ``True``, do not download any files, even if they are not present.
        Default is ``False``.
    force_download : `bool`
        If ``True``, download all files, and overwrite any existing ones.
        Default is ``False``.

    Returns
    -------
    `list` of `pathlib.Path`
        List of disk locations corresponding to the list of filenames.  An entry
        will be ``None`` if ``no_download == True`` and the file is not present.

    Raises
    ------
    RuntimeError
        Raised if any of the files cannot be downloaded from any of the mirrors.

    """
    sampledata_dir = _get_sampledata_dir()

    fullpaths = [sampledata_dir/fn for fn in filename_list]

    if no_download:
        fullpaths = [fp if fp.exists() else None for fp in fullpaths]
    else:
        to_download = zip(filename_list, fullpaths, strict=False)
        if not force_download:
            to_download = [(fn, fp) for fn, fp in to_download if not fp.exists()]

        if to_download:
            results = _download_sample_data(_BASE_URLS, to_download, overwrite=force_download)

            # Try the other mirrors for any download errors
            if results.errors:
                _handle_final_errors(results)
                raise RuntimeError

    return fullpaths


# Add a table row to the module docstring for each sample file
for _keyname, _filename in sorted(_SAMPLE_DATA.items()):
    __doc__ += f"   * - ``{_keyname}``\n     - {_filename}\n"


# file_dict and file_list are not normal variables; see __getattr__() below
__all__ = sorted(_SAMPLE_DATA.keys()) + ["download_all", "file_dict", "file_list"]  # noqa: F822, PLE0605, RUF005


# See PEP 562 (https://peps.python.org/pep-0562/) for module-level __dir__()
def __dir__():  # noqa: ANN202
    return __all__


# See PEP 562 (https://peps.python.org/pep-0562/) for module-level __getattr__()
def __getattr__(name):  # noqa: ANN001, ANN202
    if name in _SAMPLE_DATA:
        return _get_sample_files([_SAMPLE_DATA[name]])[0]
    if name == "file_dict":
        return dict(sorted(zip(_SAMPLE_DATA.keys(),
                               _get_sample_files(_SAMPLE_DATA.values(), no_download=True), strict=False)))
    if name == "file_list":
        return [v for v in __getattr__("file_dict").values() if v]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def download_all(force_download:bool=False) -> list[Path]:
    """
    Download all sample data at once that has not already been downloaded.

    Parameters
    ----------
    force_download : `bool`
        If ``True``, files are downloaded even if they already exist.  Default is
        ``False``.

    Returns
    -------
    `list` of `pathlib.Path`
        List of filepaths for sample data

    """
    return _get_sample_files(_SAMPLE_DATA.values(), force_download=force_download)
