import pytest

from punchbowl.data import sample

FILENAME = sample._SAMPLE_DATA["PUNCH_CAM"]
BAD_URL = "http://127.0.0.1:1/"


def test_download_primary(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "_get_sampledata_dir", lambda: tmp_path)

    [path] = sample._get_sample_files([FILENAME])
    assert path.exists()


def test_download_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "_get_sampledata_dir", lambda: tmp_path)
    monkeypatch.setattr(sample, "_BASE_URL", BAD_URL)

    [path] = sample._get_sample_files([FILENAME])
    assert path.exists()


def test_download_allfail(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "_get_sampledata_dir", lambda: tmp_path)
    monkeypatch.setattr(sample, "_BASE_URL", BAD_URL)
    monkeypatch.setattr(sample, "_FALLBACK_URL", BAD_URL)

    with pytest.raises(RuntimeError):
        sample._get_sample_files([FILENAME])


def test_sdac_path():
    assert sample._sdac_path(FILENAME) == f"3/CAM/2026/05/24/{FILENAME}"


def test_filenames_parse():
    for name in sample._SAMPLE_FILES:
        assert len(name.removesuffix(".fits").split("_")) == 5


def test_unknown_attribute():
    with pytest.raises(AttributeError):
        sample.CECI_NEST_PAS_UN_SAMPLE_FILE


def test_no_download_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "_get_sampledata_dir", lambda: tmp_path)
    assert sample._get_sample_files([FILENAME], no_download=True) == [None]


def test_file_list_and_dict(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "_get_sampledata_dir", lambda: tmp_path)
    (tmp_path / FILENAME).touch()

    # file_dict reports only files already on disk; file_list flattens the sequence entry
    assert sample.file_dict["PUNCH_CAM"] == tmp_path / FILENAME
    assert sample.file_dict["PUNCH_PAM"] is None
    assert len(sample.file_list) == len(sample._SAMPLE_FILES) + 1  # CAM appears twice


def test_cached_file_not_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(sample, "_get_sampledata_dir", lambda: tmp_path)
    monkeypatch.setattr(sample, "_BASE_URL", BAD_URL)
    monkeypatch.setattr(sample, "_FALLBACK_URL", BAD_URL)
    (tmp_path / FILENAME).touch()

    # would raise if it tried to download
    assert sample._get_sample_files([FILENAME]) == [tmp_path / FILENAME]
