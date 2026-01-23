import os

import numpy as np
from ndcube import NDCube

from punchbowl.data import load_ndcube_from_fits
from punchbowl.pipe.control.cache_layer import manager
from punchbowl.pipe.control.cache_layer.loader_base_class import LoaderABC
from punchbowl.util import DataLoader


class NFIL1Loader(LoaderABC):
    def __init__(self, path: str):
        self.path = path

    def load(self) -> tuple:
        with manager.try_read_from_key(self.gen_key()) as buffer:
            if buffer is None:
                cube = self.load_from_disk()
                data, meta, wcs, uncertainty_is_inf = cube.data, cube.meta, cube.wcs, np.isinf(cube.uncertainty.array)
                self.try_caching((data, meta, wcs, np.packbits(uncertainty_is_inf)))
            else:
                data, meta, wcs, uncertainty_is_inf = self.from_bytes(buffer.data)
                uncertainty_is_inf = np.unpackbits(uncertainty_is_inf, count=data.size).astype(bool).reshape(data.shape)
        return data, meta, wcs, uncertainty_is_inf

    def gen_key(self) -> str:
        return f"nfi_l1-{os.path.basename(self.path)}-{os.path.getmtime(self.path)}"

    def src_repr(self) -> str:
        return self.path

    def load_from_disk(self) -> NDCube:
        cube = load_ndcube_from_fits(self.path, include_uncertainty=True, include_provenance=False)
        return cube

    def __repr__(self):
        return f"FitsFileLoader({self.path})"


def wrap_if_appropriate(file_path: str) -> str | DataLoader:
    if manager.caching_is_enabled():
        return NFIL1Loader(file_path)
    return file_path
