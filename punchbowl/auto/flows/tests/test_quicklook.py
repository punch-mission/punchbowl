import pytest

from punchbowl.auto.flows.quicklook import (
    generate_quicklook_mzp,
    generate_quicklook_tbpb,
    generate_quicklook_unpolarized,
)
from punchbowl.data.tests.test_punch_io import sample_ndcube


def test_quicklook_unpolarized(sample_ndcube, tmp_path):
    cube = sample_ndcube(shape=(10, 10), code="CAM", level="3")
    generate_quicklook_unpolarized(cube, file_list=[],
                                   make_image=True, make_movie=False,
                                   output_quicklook_dir=str(tmp_path),
                                   framerate=10, ffmpeg_command="ffmpeg")

    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_tB_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}.jpg").exists()
    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_tB_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}_thumbnail.jpg").exists()


def test_quicklook_tbpb(sample_ndcube, tmp_path):
    cube = sample_ndcube(shape=(2, 10, 10), code="PAM", level="3")
    generate_quicklook_tbpb(cube, file_list=[],
                                   make_image=True, make_movie=False,
                                   output_quicklook_dir=str(tmp_path),
                                   framerate=10, ffmpeg_command="ffmpeg")

    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_tB_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}.jpg").exists()
    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_pB_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}.jpg").exists()
    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_tB_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}_thumbnail.jpg").exists()
    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_pB_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}_thumbnail.jpg").exists()


def test_quicklook_mzp(sample_ndcube, tmp_path):
    cube = sample_ndcube(shape=(3, 10, 10), code="PIM", level="3")
    generate_quicklook_mzp(cube, file_list=[],
                                   make_image=True, make_movie=False,
                                   output_quicklook_dir=str(tmp_path),
                                   framerate=10, ffmpeg_command="ffmpeg")

    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_M_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}.jpg").exists()
    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_Z_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}.jpg").exists()
    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_P_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}.jpg").exists()
    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_M_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}_thumbnail.jpg").exists()
    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_Z_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}_thumbnail.jpg").exists()
    assert (tmp_path / f"PUNCH_{cube.meta['TYPECODE'].value}{cube.meta['OBSCODE'].value}_P_{cube.meta.datetime.strftime("%Y%m%d")}_v{cube.meta['FILEVRSN'].value}_thumbnail.jpg").exists()
